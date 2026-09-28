# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_acquisition.py
"""ADP-009D: planner metadata reads own one bounded, retained acquisition."""
import json
import os

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def module():
    from blueprint_pipeline import task_evaluation_scene_lifecycle_acquisition as acquisition
    return acquisition


def test_exact_metadata_read_and_final_named_identity(tmp_path):
    m = module()
    path = tmp_path / 'owner.json'
    raw = b'{"schema_version":"tiny.v1"}\n'
    path.write_bytes(raw)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    with m.Acquisition(budget, [str(tmp_path)]) as reader:
        assert reader.read_json(str(path)) == raw
        assert reader.verify() is True
        replacement = tmp_path / 'replacement.json'
        replacement.write_bytes(raw)
        os.replace(replacement, path)
        with pytest.raises(m.AcquisitionError, match='metadata_changed'):
            reader.verify()
    assert budget.counts['roots'] == 1
    assert budget.counts['raw_bytes'] == len(raw)


def test_payload_never_opened_and_linked_metadata_refuses(tmp_path, monkeypatch):
    m = module()
    payload = tmp_path / 'payload.bin'
    payload.write_bytes(b'opaque')
    linked = tmp_path / 'alias.json'
    linked.symlink_to(payload)
    with m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)]) as reader:
        with pytest.raises(m.AcquisitionError):
            reader.read_json(str(payload))
        with pytest.raises(m.AcquisitionError):
            reader.read_json(str(linked))
        monkeypatch.setattr(m.os, 'open', lambda *a, **k: pytest.fail('payload opened'))
        info = reader.stat(str(payload))
        assert info.st_size == 6


def test_shared_read_allowance_refuses_before_next_read(tmp_path, monkeypatch):
    m = module()
    path = tmp_path / 'owner.json'
    path.write_text('{}')
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('raw_bytes', budget.limits['raw_bytes'])
    with m.Acquisition(budget, [str(tmp_path)]) as reader:
        monkeypatch.setattr(m.os, 'read', lambda *a: pytest.fail('read past shared cap'))
        with pytest.raises(ValueError, match='reference_raw_bytes_limit'):
            reader.read_json(str(path))


def test_listing_counts_before_retention_and_checks_names(tmp_path, monkeypatch):
    m = module()
    (tmp_path / 'owner.json').write_text('{}')
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    with m.Acquisition(budget, [str(tmp_path)]) as reader:
        assert reader.entries(str(tmp_path)) == ('owner.json',)
        assert budget.counts['entries'] >= 1
        (tmp_path / 'other.json').write_text(json.dumps({'a': 1}))
        with pytest.raises(m.AcquisitionError, match='metadata_changed'):
            reader.verify()


def test_path_component_and_anchor_caps_precede_open(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m.os, 'open', lambda *a, **k: pytest.fail('unsafe path opened'))
    with pytest.raises(m.AcquisitionError):
        m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path / '..')])
    with pytest.raises(m.AcquisitionError):
        m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)] * 5)


@pytest.mark.parametrize('repeat', ['stat', 'read_json'])
def test_repeated_observation_cannot_refresh_first_raw_identity(tmp_path, repeat):
    m = module()
    file = tmp_path / 'owner.json'
    file.write_bytes(b'{"version":1}')
    with m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)]) as reader:
        assert reader.read_json(str(file)) == b'{"version":1}'
        file.write_bytes(b'{"version":2}')  # Same inode and size; first bytes stay bound.
        with pytest.raises(m.AcquisitionError, match='metadata_changed'):
            getattr(reader, repeat)(str(file))


def test_repeated_listing_cannot_refresh_first_membership(tmp_path):
    m = module()
    with m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)]) as reader:
        assert reader.entries(str(tmp_path)) == ()
        (tmp_path / 'new.json').write_bytes(b'{}')
        with pytest.raises(m.AcquisitionError, match='metadata_changed'):
            reader.entries(str(tmp_path))


def test_recursive_directory_cap_is_checked_before_each_open(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'MAX_DIRECTORIES', 1)
    real_open, calls = m.os.open, []
    def observed_open(*args, **kwargs):
        calls.append(args[0])
        return real_open(*args, **kwargs)
    monkeypatch.setattr(m.os, 'open', observed_open)
    with pytest.raises(m.AcquisitionError, match='directories_limit'):
        m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)])
    assert calls == ['/']


def test_final_root_named_identity_is_rechecked(tmp_path, monkeypatch):
    from types import SimpleNamespace
    m = module()
    with m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)]) as reader:
        original = m.os.stat
        def changed(name, *args, **kwargs):
            info = original(name, *args, **kwargs)
            if name == '/':
                return SimpleNamespace(**{key: getattr(info, key) + (1 if key == 'st_ino' else 0)
                                         for key in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')})
            return info
        monkeypatch.setattr(m.os, 'stat', changed)
        with pytest.raises(m.AcquisitionError, match='metadata_changed'):
            reader.verify()


def test_final_retained_directory_descriptor_identity_is_rechecked(tmp_path, monkeypatch):
    from types import SimpleNamespace
    m = module()
    with m.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), [str(tmp_path)]) as reader:
        original = m.os.fstat
        victim = reader.directories[str(tmp_path)][0]
        def changed(fd):
            info = original(fd)
            if fd == victim:
                return SimpleNamespace(**{key: getattr(info, key) + (1 if key == 'st_ino' else 0)
                                         for key in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')})
            return info
        with monkeypatch.context() as scoped:
            scoped.setattr(m.os, 'fstat', changed)
            with pytest.raises(m.AcquisitionError, match='metadata_changed'):
                reader.verify()
