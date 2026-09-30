"""Current historical reference tables; no deletion authority follows."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_references.py
import os
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def scan(root, target):
    from blueprint_pipeline.control_plane_lane_historical_references import _table
    budget = ReferenceCollectionBudget()
    try:
        return _table(Path(root), Path(target), budget)
    finally:
        budget.close()


def test_pending_row_naming_generation_is_a_reference(tmp_path):
    queue = tmp_path / 'queue'
    (queue / 'pending').mkdir(parents=True)
    target = tmp_path / 'selected'
    (queue / 'pending/job.json').write_text('{"source":"' + str(target / 'input.bin') + '"}')
    with pytest.raises(ValueError, match='table_reference'):
        scan(queue, target)


def test_closed_unchanged_tables_bind_exact_namespace_and_bytes(tmp_path):
    (tmp_path / 'processing').mkdir()
    (tmp_path / 'processing/job.json').write_text('{"run":"other"}')
    before = scan(tmp_path, Path('/unrelated/selected'))
    assert len(before) == 3
    (tmp_path / 'processing/job.json').write_text('{"run":"changed"}')
    assert scan(tmp_path, Path('/unrelated/selected')) != before


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'malformed', 'fifo'])
def test_unknown_reference_rows_are_not_absence(tmp_path, kind):
    row = tmp_path / 'job.json'
    if kind == 'symlink':
        row.symlink_to(tmp_path / 'absent')
    elif kind == 'hardlink':
        row.write_text('{}')
        os.link(row, tmp_path / 'alias.json')
    elif kind == 'fifo':
        os.mkfifo(row)
    else:
        row.write_text('{bad json')
    with pytest.raises(ValueError, match='table_unknown'):
        scan(tmp_path, Path('/unrelated/selected'))
