# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_downstream_contracts.py
"""Exact private JSON measurement stays bounded during long strings."""
import json

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from blueprint_pipeline.task_evaluation_scene_downstream_contracts import bounded_size


@pytest.mark.parametrize('value', [
    'a' * 8193, ('\"\\\b\f\n\r\t\x00é中😀' * 700),
    {'long': ['é' * 1023 + '😀' + '\\', True, None, -12, 0.5]},
])
def test_private_measurement_keeps_exact_utf8_and_escape_lengths(value):
    expected = len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode())
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    assert bounded_size(value, 128 * 1024, work_budget=budget) == expected


@pytest.mark.parametrize('value', ['a' * 1024 + '\ud800', '\udfff' + 'b' * 2048])
def test_private_measurement_rejects_surrogates_across_chunk_boundaries(value):
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    with pytest.raises(ValueError):
        bounded_size(value, 128 * 1024, work_budget=budget)


def test_private_measurement_checks_deadline_after_each_bounded_encoder_call(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as contracts
    clock = [0.0]
    budget = ReferenceCollectionBudget(monotonic=lambda: clock[0])
    original = contracts.json.dumps
    sizes = []
    def observed(value, **kwargs):
        sizes.append(len(value))
        result = original(value, **kwargs)
        clock[0] = 6.0
        return result
    monkeypatch.setattr(contracts.json, 'dumps', observed)
    with pytest.raises(ValueError, match='reference_deadline_exceeded'):
        bounded_size('a' * 8193, 128 * 1024, work_budget=budget)
    assert len(sizes) == 1 and sizes[0] <= 1024


def test_private_measurement_refuses_oversized_string_before_encoding(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as contracts
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    monkeypatch.setattr(contracts.json, 'dumps', lambda *args, **kwargs: pytest.fail('encoder past cap'))
    with pytest.raises(ValueError):
        bounded_size('a' * 1025, 1024, work_budget=budget)
