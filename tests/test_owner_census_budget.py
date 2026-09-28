"""Opt-in consent composition shares one budget while the old validator stays pure."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_decisions.py
#   src/blueprint_pipeline/control_plane_reference_budget.py

import os

import pytest

from blueprint_pipeline import control_plane_lane_scratch_decisions as d
from blueprint_pipeline import control_plane_reference_budget as b
from tests.test_lane_scratch_decisions import _annotations, _inventory, _json


def payloads():
    row = dict(path='/work/sample', family='other', owner_guess='unowned',
               owner_guess_basis='no_owner_evidence', allocated_bytes=4,
               newest_mtime_epoch=None, age_seconds=None, unreadable=0,
               shared_names=0, references=[], owner_decision=None, approved_expiry=None)
    census = _json(_inventory([row]))
    annotations = _annotations(census, [dict(path=row['path'], action='keep', owner='owner', expires_at_epoch=1100)])
    return census, annotations


def test_budget_validator_matches_ordinary_result_and_never_accesses_targets(monkeypatch):
    census, annotations = payloads()
    expected = d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',))
    original_stat = os.stat
    def guarded_stat(path, *args, **kwargs):
        if str(path).startswith(('/work', '/inputs')):
            pytest.fail('target stat')
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(d.os, 'stat', guarded_stat)
    shared = b.ReferenceCollectionBudget(monotonic=lambda: 0)
    actual = d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',), _work_budget=shared)
    assert actual == expected
    assert shared.counts['raw_bytes'] == len(census) + len(annotations)
    assert shared.counts['rows'] >= 1 and shared.counts['facts'] >= 1
    assert shared.counts['values'] > 0 and shared.counts['output_bytes'] > 0
    assert not shared.closed


def test_combined_byte_cap_refuses_before_either_parser_or_hash(monkeypatch):
    census, annotations = payloads()
    monkeypatch.setattr(b, 'MAX_RAW_BYTES', len(census) + len(annotations) - 1)
    monkeypatch.setattr(d.json, 'loads', lambda *a, **k: pytest.fail('parser'))
    monkeypatch.setattr(d.hashlib, 'sha256', lambda *a, **k: pytest.fail('hash'))
    with pytest.raises(b.ReferenceCollectionBudgetError, match='reference_raw_bytes_limit'):
        d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',),
                                     _work_budget=b.ReferenceCollectionBudget(monotonic=lambda: 0))


def test_both_documents_lexical_preflight_precedes_first_parser(monkeypatch):
    census, _ = payloads()
    annotations = b'{"nested":' + b'['*8 + b'0' + b']'*8 + b'}'
    monkeypatch.setattr(b, 'MAX_DEPTH', 4)
    monkeypatch.setattr(d.json, 'loads', lambda *a, **k: pytest.fail('parsed first doc before second preflight'))
    with pytest.raises(b.ReferenceCollectionBudgetError, match='reference_depth_limit'):
        d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',),
                                     _work_budget=b.ReferenceCollectionBudget(monotonic=lambda: 0))


def test_budget_refuses_decoded_output_before_encoder(monkeypatch):
    census, annotations = payloads()
    monkeypatch.setattr(b, 'MAX_OUTPUT_BYTES', 8)
    monkeypatch.setattr(d.json, 'dumps', lambda *a, **k: pytest.fail('unbounded encoder'))
    with pytest.raises(b.ReferenceCollectionBudgetError, match='reference_output_bytes_limit'):
        d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',),
                                     _work_budget=b.ReferenceCollectionBudget(monotonic=lambda: 0))


def test_none_path_keeps_independent_native_input_limits(monkeypatch):
    census, annotations = payloads()
    monkeypatch.setattr(b, 'MAX_RAW_BYTES', 1)
    monkeypatch.setattr(b.ReferenceCollectionBudget, 'tick', lambda self: pytest.fail('None touched B'))
    assert d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',),
                                       _work_budget=None)['status'] == 'validated'


def test_enabled_register_keeps_protocol_omission(monkeypatch):
    census, _ = payloads()
    decision = dict(path='/work/sample', action='register', owner='owner', lane='g1', name='sample',
                    reason='test-cache', class_intent='cache', cleanup='owner_review',
                    ttl_seconds=100, run_ref='run-1', size_budget_bytes=4)
    annotations = _annotations(census, [decision])
    seen = []
    original = d._creation_lease
    def create(**kwargs):
        assert 'consumer_lifetime_contract' not in kwargs
        seen.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(d, '_creation_lease', create)
    result = d.validate_census_annotations(census, annotations, now=1000, allowed_roots=('/work',),
                                         _work_budget=b.ReferenceCollectionBudget(monotonic=lambda: 0))
    assert seen and result['execution_authorized'] is False


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":"\\ud800"}'])
def test_shared_malformed_json_is_still_typed(raw):
    with pytest.raises((d.CensusDecisionError, b.ReferenceCollectionBudgetError)):
        d.validate_census_annotations(raw, b'{}', now=1000, allowed_roots=('/work',),
                                     _work_budget=b.ReferenceCollectionBudget(monotonic=lambda: 0))
