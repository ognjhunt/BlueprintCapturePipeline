"""Real owned descriptor work keeps metadata phases separate from payload IO."""
import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_retirement_flow import retirement_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401


def test_long_payload_keeps_old_budget_closed_and_original_fd_guards(tmp_path):
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    clock = [0.0]
    root = tmp_path/'target'
    root.mkdir()
    (root/'member').write_bytes(b'tiny payload')
    files = _ActionFiles(monotonic=lambda: clock[0], now=lambda: 1000)
    try:
        target, _ = files.parent(root/'member')
        original = files.budget
        files.payload(root, target, expected_payload_bytes=12)
        assert original.closed and original.failure is None
        clock[0] = 6.0
        member = files.open('member', 0, parent=target)
        files.location(member)
        assert files.payload_read(member, 12, role='issue_hash') == b'tiny payload'
        files.close(member)
        files.phase('finalize')
        assert files.budget is not original and original.closed and original.failure is None
        files.verify()
        assert files.controller_origin == 0.0
    finally:
        files.finish()
        files.budget.close()


def test_payload_refusal_is_sticky_and_cannot_be_reset_with_new_phase(tmp_path):
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    clock = [0.0]
    root = tmp_path/'target'
    root.mkdir()
    (root/'member').write_bytes(b'tiny')
    files = _ActionFiles(monotonic=lambda: clock[0], now=lambda: 1000)
    try:
        fd, _ = files.parent(root/'member')
        files.payload(root, fd, expected_payload_bytes=4)
        clock[0] = 4*3600+1
        with pytest.raises(ValueError, match='experiment_'):
            files.location(fd)
        with pytest.raises(ValueError, match='experiment_'):
            files.phase('finalize')
    finally:
        files.finish()
        files.budget.close()


def test_declared_metadata_phase_cannot_be_repeated_to_extend_deadline(tmp_path):
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    files = _ActionFiles()
    try:
        files.phase('manifest')
        first = files.budget
        with pytest.raises(ValueError, match='experiment_'):
            files.phase('manifest')
        assert files.budget is first
    finally:
        files.finish()
        files.budget.close()


def test_immutable_event_publication_releases_only_its_original_fds(tmp_path, root_metadata):  # noqa: F811
    import json
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    destination = tmp_path/'events'
    destination.mkdir(mode=0o700)
    files = _ActionFiles()
    try:
        parent, _ = files.parent(destination/'e-00000.json', protected=True)
        original = set(files.owned)
        for index in range(32):
            raw = json.dumps({'index': index}).encode()
            selected = actions._publish(files, parent, f'e-{index:05d}.json', raw, kind='event')
            assert selected['size_bytes'] == len(raw)
            assert set(files.owned) == original
        assert len(tuple(destination.iterdir())) == 32
    finally:
        files.finish()
        files.budget.close()


def test_actual_gc_uses_finite_sixteen_member_removal_batches(retirement_installation, monkeypatch):  # noqa: F811
    from tests.test_registered_experiment_retirement_flow import _born_scratch, _issue_action, _gc
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    grant, _, target = _born_scratch(retirement_installation)
    for index in range(32):
        (target/f'member-{index:02d}').write_bytes(b'x')
    action = _issue_action(retirement_installation, grant)
    phases = []
    original = _ActionFiles.phase
    def observe(files, name):
        phases.append(name)
        return original(files, name)
    monkeypatch.setattr(_ActionFiles, 'phase', observe)
    report = _gc(retirement_installation)
    outcome = next(row for row in report['registered_experiments']['outcomes'] if row['action_id'] == action['action_id'])
    assert outcome['decision'] == 'retired', outcome
    assert phases.count('removal_batch') == 3
    assert {path.name for path in target.iterdir()} == {'.lane-scratch.v1.json', '.registered-experiment.v1.json'}


def test_same_payload_role_cannot_rewind_and_refund_its_source_window(tmp_path):
    import os
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    root = tmp_path/'target'
    root.mkdir()
    (root/'member').write_bytes(b'tiny')
    files = _ActionFiles()
    try:
        target, _ = files.parent(root/'member')
        files.payload(root, target, expected_payload_bytes=4)
        fd = files.open('member', os.O_RDONLY, parent=target)
        assert files.payload_read(fd, 4, role='issue_hash') == b'tiny'
        os.lseek(fd, 0, os.SEEK_SET)
        with pytest.raises(ValueError, match='experiment_work_payload_cursor_changed'):
            files.payload_read(fd, 4, role='issue_hash')
        with pytest.raises(ValueError, match='experiment_work_'):
            files.phase('finalize')
    finally:
        files.finish()
        files.budget.close()
