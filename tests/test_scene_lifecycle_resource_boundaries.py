# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_pool.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_acquisition.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_cli.py
"""Tiny resource refusals precede the next parser, hash, read or expansion."""
import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def pool(raw):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_pool import Pool
    class Reader:
        budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    value = Pool(Reader(), {}, 'intent-1')
    value.raw = [('opaque_evidence', '/metadata/'+str(i)+'.json', row) for i, row in enumerate(raw)]
    return value


def test_all_raw_preflight_precedes_first_parse_or_hash(monkeypatch):
    from blueprint_pipeline import control_plane_reference_budget as budget
    from blueprint_pipeline import task_evaluation_scene_lifecycle_pool as module
    value = pool([b'{"a":0}', b'{"a":[]}'])
    monkeypatch.setattr(budget, 'MAX_DEPTH', 1)
    monkeypatch.setattr(module.json, 'loads', lambda *a, **k: pytest.fail('first record parsed before all lexical bounds'))
    monkeypatch.setattr(module.hashlib, 'sha256', lambda *a, **k: pytest.fail('proof hashed before all lexical bounds'))
    with pytest.raises(ValueError, match='reference_depth_limit'):
        value.decode()


def test_actual_decoded_depth_is_checked_after_parse_before_hash(monkeypatch):
    from blueprint_pipeline import control_plane_reference_budget as budget
    from blueprint_pipeline import task_evaluation_scene_lifecycle_pool as module
    value = pool([b'{"a":0}'])
    monkeypatch.setattr(budget, 'MAX_DEPTH', 1)
    original, calls = module.json.loads, []
    def parsed(*a, **k):
        calls.append(1)
        assert original(*a, **k) == {'a': 0}
        # Inject the decoder boundary; lexical input bounds cannot substitute
        # for checking the actual value before proof hashing.
        return {'a': {'b': 0}}
    monkeypatch.setattr(module.json, 'loads', parsed)
    monkeypatch.setattr(module.hashlib, 'sha256', lambda *a, **k: pytest.fail('proof hashed before actual scalar depth'))
    with pytest.raises(ValueError, match='reference_depth_limit'):
        value.decode()
    assert calls == [1]


def test_noncoalesced_context_parent_consumes_four_anchor_allowance(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
    from pathlib import Path
    anchors = []
    for i in range(4):
        child = tmp_path/str(i)
        child.mkdir()
        anchors.append(str(child))
    shared = ReferenceCollectionBudget(monotonic=lambda: 0)
    with Acquisition(shared, [str(tmp_path)]) as reader:
        with pytest.raises(ValueError, match='anchors_limit'):
            reader.add_planner_anchors(anchors)
        assert len({identity for _, identity in reader.anchors}) == 4
    assert not reader.handles
    assert all(Path(name).is_dir() for name in anchors)


def test_exact_same_context_parent_coalesces_once_and_retains_named_identity(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
    shared = ReferenceCollectionBudget(monotonic=lambda: 0)
    with Acquisition(shared, [str(tmp_path)]) as reader:
        assert reader.add_planner_anchors([str(tmp_path)]) is True
        assert shared.counts['roots'] == 1
        assert reader.verify()


def test_entry_exhaustion_does_not_publish_partial_membership(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
    from blueprint_pipeline import control_plane_reference_budget as module
    (tmp_path/'a.json').write_text('{}')
    (tmp_path/'b.json').write_text('{}')
    monkeypatch.setattr(module, 'MAX_ENTRIES', 1)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    with Acquisition(budget, [str(tmp_path)]) as reader:
        with pytest.raises(ValueError, match='reference_entries_limit'):
            reader.entries(str(tmp_path))
        assert str(tmp_path) not in reader.memberships
        assert budget.counts['entries'] == 1
    assert not reader.handles


def test_expired_clock_refuses_next_metadata_operation_but_owned_cleanup_continues(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_acquisition as module
    current = [0]
    budget = ReferenceCollectionBudget(monotonic=lambda: current[0])
    with module.Acquisition(budget, [str(tmp_path)]) as reader:
        current[0] = 10
        monkeypatch.setattr(module.os, 'stat', lambda *a, **k: pytest.fail('metadata used after expiry'))
        with pytest.raises(ValueError, match='reference_deadline_exceeded'):
            reader.stat(str(tmp_path/'owner.json'))
    assert not reader.handles


def test_initial_descriptor_identity_failure_never_closes_unowned_numeric_token(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_acquisition as module
    closed = []
    monkeypatch.setattr(module.os, 'open', lambda *a, **k: 1234567)
    def unknown(fd):
        raise OSError('unavailable identity')
    monkeypatch.setattr(module.os, 'fstat', unknown)
    monkeypatch.setattr(module.os, 'close', closed.append)
    with pytest.raises(ValueError, match='descriptor_ownership_unproven'):
        module.Acquisition(ReferenceCollectionBudget(monotonic=lambda: 0), ['/metadata'])
    assert not closed
