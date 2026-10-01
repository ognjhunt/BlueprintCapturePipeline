"""ADP-009D/day28: allocation rehearsal precedes private snapshot effects.

These pure shape tests project checkpoint costs; they prove no native execution.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from blueprint_pipeline.control_plane_lane_historical_authority import _HistoricalFiles
from blueprint_pipeline.control_plane_lane_historical_restore_metadata import preflight_restore_metadata
from blueprint_pipeline.control_plane_lane_historical_restore_metadata import strict_document_value_work
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


@pytest.mark.parametrize('members,admitted', [(100, True), (2000, False)])
def test_snapshot_shape_and_final_selection_share_the_unchanged_budget(members, admitted):
    original = {'members': [{'path': str(index), 'version': list(range(10))}
                            for index in range(members)]}
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    calls = []
    @contextmanager
    def checkpoint(*, journal):
        assert journal is True
        # Actual allocations model the pre/post selection work, not an owner
        # decision or a kernel observation. The body cannot reset this budget.
        budget.measure(list(range(20_000)))
        calls.append('selected')
        yield files, None, None
        budget.measure(list(range(20_000)))
        calls.append('reselected')
    worker = SimpleNamespace(selected=(None, None, original), checkpoint=checkpoint)
    try:
        if admitted:
            preflight_restore_metadata(worker)
            assert calls == ['selected', 'reselected']
        else:
            with pytest.raises(ValueError, match='reference_values_limit'):
                preflight_restore_metadata(worker)
            assert budget.failure == 'reference_values_limit'
        assert budget.limits['values'] == 100_000
        assert files.document_nodes == {}  # strict snapshot path stays uncached
    finally:
        files.finish()
        budget.close()


@pytest.mark.parametrize('value', [
    {'one': 1}, {'a': [1, {'b': 'escaped\x01path'}], 'c': None},
    {'version': list(range(10)), 'selector': {'sha256': 'sha256:' + 'a' * 64, 'size_bytes': 7}},
])
def test_strict_work_reservation_matches_the_actual_bounded_parser(value):
    import json
    from blueprint_pipeline import control_plane_lane_scratch_decisions as retained
    budget = ReferenceCollectionBudget()
    raw = json.dumps(value, separators=(',', ':')).encode()
    expected = strict_document_value_work(value, budget.tick)
    retained._document(raw, 1000, _work_budget=budget)
    assert budget.counts['values'] == expected
    budget.close()
