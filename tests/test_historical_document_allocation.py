"""ADP-009D/day28: intrinsic JSON proofs never replace current authority.

Pure allocation accounting; these tests grant no native or owner authority.
"""
import pytest

from blueprint_pipeline.control_plane_lane_historical_authority import _HistoricalFiles
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def test_identical_bytes_allocate_fresh_values_under_the_original_budget():
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    raw = b'{"a":[1,2],"b":3}'
    try:
        first = files.document(raw, cap=100)
        before = budget.counts['values']
        first['a'].append('caller mutation')
        second = files.document(raw, cap=100)
        assert second == {'a': [1, 2], 'b': 3}
        assert budget.counts['values'] - before == 7
        assert budget.limits['values'] == 100_000
    finally:
        files.finish()
        budget.close()


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'{broken'])
def test_changed_or_invalid_bytes_need_the_complete_strict_proof(raw):
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    try:
        files.document(b'{"a":1}', cap=100)
        with pytest.raises(ValueError):
            files.document(raw, cap=100)
    finally:
        files.finish()
        budget.close()


def test_current_cap_and_exhaustion_precede_cached_allocation(monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    raw = b'{"a":1}'
    try:
        files.document(raw, cap=100)
        with pytest.raises(ValueError):
            files.document(raw, cap=3)
        budget.charge('values', budget.limits['values'] - budget.counts['values'])
        def forbidden(*args, **kwargs):
            raise AssertionError('allocation occurred after exhaustion')
        monkeypatch.setattr(authority.json, 'loads', forbidden)
        with pytest.raises(ValueError, match='reference_values_limit'):
            files.document(raw, cap=100)
    finally:
        files.finish()
        budget.close()


def test_allocation_proofs_expire_with_the_original_files_lifetime():
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    files.document(b'{"a":1}', cap=100)
    assert files.document_nodes
    files.finish()
    assert files.document_nodes == {}
    budget.close()


def test_cached_canonical_size_still_obeys_a_lower_current_cap():
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    raw = b'{"x":1e3}'
    try:
        files.document(raw, cap=100)
        with pytest.raises(ValueError):
            files.document(raw, cap=len(raw))
    finally:
        files.finish()
        budget.close()


def test_changed_valid_bytes_get_a_new_proof_and_value():
    budget = ReferenceCollectionBudget()
    files = _HistoricalFiles(budget)
    try:
        assert files.document(b'{"x":1}', cap=100) == {'x': 1}
        assert files.document(b'{"x":2}', cap=100) == {'x': 2}
        assert len(files.document_nodes) == 2
    finally:
        files.finish()
        budget.close()
