"""Only strictly longer prefixes can replace the already validated winner."""
import pytest
from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from blueprint_pipeline.task_evaluation_sam31_prefix_billing import Sam31PrefixBillingPending


def test_lower_prefixes_do_not_repeat_scientific_validation(monkeypatch):
    calls = []
    def validate(**kw):
        phase = kw['through_phase']
        calls.append(phase)
        assert adoption.PREFIX_LENGTHS[phase] > 5
        raise ValueError('incomplete_tail')
    monkeypatch.setattr(adoption, 'materialize_completed_prefix_adoption', validate)
    result = adoption.select_completed_prefix_adoption(minimum_prefix_length=5)
    assert result['status'] == 'no_reusable_prefix'
    assert calls == [p for p in reversed(adoption.PREFIX_LENGTHS) if adoption.PREFIX_LENGTHS[p] > 5]
    skipped = [r for r in result['rejected_candidates'] if r['blocker'] == 'prefix_not_longer_than_verified_selection']
    assert {r['through_phase'] for r in skipped} == {p for p, n in adoption.PREFIX_LENGTHS.items() if n <= 5}


def test_longer_prefix_still_requires_full_validation_and_billing(monkeypatch):
    calls = []
    def validate(**kw):
        phase = kw['through_phase']
        calls.append(phase)
        if phase == 'contribution_sweep':
            raise Sam31PrefixBillingPending('sam31_adoption_official_billing_pending')
        raise ValueError('incomplete_tail')
    monkeypatch.setattr(adoption, 'materialize_completed_prefix_adoption', validate)
    with pytest.raises(Sam31PrefixBillingPending):
        adoption.select_completed_prefix_adoption(minimum_prefix_length=5)
    assert calls[-1] == 'contribution_sweep'


def test_longer_valid_prefix_can_replace_winner(monkeypatch):
    calls = []
    def validate(**kw):
        calls.append(kw['through_phase'])
        if kw['through_phase'] != 'contribution_sweep':
            raise ValueError('incomplete_tail')
        return {'validated': True}
    monkeypatch.setattr(adoption, 'materialize_completed_prefix_adoption', validate)
    result = adoption.select_completed_prefix_adoption(minimum_prefix_length=5)
    assert result['status'] == 'reusable_prefix_selected'
    assert result['through_phase'] == 'contribution_sweep'
    assert result['adoption'] == {'validated': True}


@pytest.mark.parametrize('floor', [True, -1, 100, 5.0, '5'])
def test_invalid_floor_refused(floor):
    with pytest.raises(ValueError, match='minimum_invalid'):
        adoption.select_completed_prefix_adoption(minimum_prefix_length=floor)
