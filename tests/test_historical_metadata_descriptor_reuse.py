"""ADP-009D/day28: fresh reads retain exact original metadata descriptors.

These ordinary-UID IO tests grant no protected owner or native authority.
"""
import os

import pytest

from blueprint_pipeline.control_plane_lane_historical_authority import _HistoricalFiles
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def test_repeated_fresh_metadata_reads_do_not_acquire_duplicate_live_descriptors(tmp_path):
    path = tmp_path / 'record.json'
    path.write_bytes(b'{"value":1}')
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    try:
        raw, original = files.read(path, cap=100)
        count = len(files.owned)
        for _ in range(128):
            current, record = files.read(path, cap=100)
            assert current == raw
        assert record == original and len(files.owned) == count
        assert budget.counts['raw_bytes'] == len(raw) * 129
        files.verify()
    finally:
        files.finish()
        budget.close()


@pytest.mark.parametrize('change', ['bytes', 'inode', 'rights', 'cap'])
def test_reused_metadata_descriptor_never_hides_current_drift(tmp_path, change):
    path = tmp_path / 'record.json'
    path.write_bytes(b'{"value":1}')
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    try:
        files.read(path, cap=100)
        if change == 'bytes':
            path.write_bytes(b'{"value":2}')
        elif change == 'inode':
            other = tmp_path / 'replacement'
            other.write_bytes(b'{"value":1}')
            os.replace(other, path)
        elif change == 'rights':
            path.chmod(0o400)
        with pytest.raises(ValueError):
            files.read(path, cap=3 if change == 'cap' else 100)
    finally:
        # The failed original records stay retained through cleanup; no fresh
        # path adoption or blind close is used to turn this into a passing proof.
        files.finish()
        budget.close()
