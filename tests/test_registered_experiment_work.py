"""Real owned descriptor work keeps metadata phases separate from payload IO."""
from pathlib import Path

import pytest


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
